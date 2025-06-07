package com.ssafy.hellojob.domain.sse.service;

import com.ssafy.hellojob.domain.sse.dto.AckRequestDto;
import com.ssafy.hellojob.global.util.JsonUtil;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Deque;
import java.util.Iterator;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;

@Service
@Slf4j
@RequiredArgsConstructor
public class SSEService {

    public record SseEventWrapper(String eventName, String dataJson) {
    }

    private final Map<Integer, Deque<SseEmitter>> emitters = new ConcurrentHashMap<>();
    private final int MAX_EMITTERS_PER_USER = 3;
    private final Map<Integer, Queue<SseEventWrapper>> retryQueue = new ConcurrentHashMap<>();
    private final JsonUtil jsonUtil;

    public void addEmitter(Integer userId, SseEmitter emitter) {
        emitters.compute(userId, (key, existingDeque ) -> {
            Deque<SseEmitter> deque = (existingDeque != null) ? existingDeque : new ConcurrentLinkedDeque<>();

            while (deque.size() >= MAX_EMITTERS_PER_USER) {
                SseEmitter old = deque.pollFirst();
                try {
                    if (old != null) old.complete(); // 이전 연결 닫기
                } catch (Exception e) {
                    log.warn("이전 emitter 종료 중 에러: {}", e.getMessage());
                }
            }

            deque.addLast(emitter);
            log.debug("🧷 emitter 추가됨 - userId={}, 현재 연결 수={}", userId, deque.size());
            return deque;
        });

        // 연결 종료 시 emitter 제거
        emitter.onCompletion(() -> {
            log.debug("SSE 연결 정상 종료");
            removeEmitter(userId, emitter);
        });
        emitter.onTimeout(() -> {
            log.debug("SSE 타임아웃으로 연결 종료");
            removeEmitter(userId, emitter);
        });
        emitter.onError(e -> {
            log.debug("SSE 연결 중 에러 발생 userId: {} | {} ", userId, e.getMessage());
            removeEmitter(userId, emitter);
        });
    }

    private void removeEmitter(Integer userId, SseEmitter emitter) {
        Deque<SseEmitter> deque = emitters.get(userId);
        if (deque != null) {
            deque.remove(emitter);
            if (deque.isEmpty())
                emitters.remove(userId);
        }
    }

    public Deque<SseEmitter> getEmitters(Integer userId) {
        return emitters.get(userId);
    }

    public void sendToUser(Integer userId, String eventName, Object data) {
        log.debug("왜 두 번 찍히는거야 !!!!!");
        log.debug("eventName: {}", eventName);

        // 일단 큐에 넣음
        queueEvent(userId, eventName, data);
        Deque<SseEmitter> emittersDeque = getEmitters(userId);
        if (emittersDeque != null) {
            Iterator<SseEmitter> iterator = emittersDeque.iterator();
            while (iterator.hasNext()) {
                SseEmitter emitter = iterator.next();
                try {
                    emitter.send(SseEmitter.event()
                            .name(eventName)
                            .data(data));
                } catch (IOException e) {
                    // 연결이 끊긴 경우
                    log.warn("❌ SSE 연결 실패 - userId={}, 원인={}", userId, e.getMessage());
                    emitter.completeWithError(e);
                    removeEmitter(userId, emitter);
                }
            }
        } else {
            log.debug("🔇 연결 없음 - userId = {}, 큐에 보관", userId);
        }
    }

    public void queueEvent(Integer userId, String eventName, Object data) {
        retryQueue
                .computeIfAbsent(userId, k -> new ConcurrentLinkedDeque<>())
                .add(new SseEventWrapper(eventName, jsonUtil.toJson(data))); // 문자열로 저장
    }

    // 클라이언트 재접속 시 큐에 저장한 event 재실행
    public void replayQueuedEvents(Integer userId, SseEmitter emitter) {
        log.debug("▶️ replayQueuedEvents 시작");
        Queue<SseEventWrapper> queue = retryQueue.get(userId);

        if (queue != null && !queue.isEmpty()) {
            log.debug("▶️ userId={}, 큐 크기={}", userId, queue.size());
            while (!queue.isEmpty()) {
                SseEventWrapper event = queue.peek();
                try {
                    emitter.send(SseEmitter.event()
                            .name(event.eventName())
                            .data(event.dataJson()));
                    queue.poll(); // 전송 성공 시에만 꺼냄
                } catch (IOException e) {
                    log.warn("❌ SSE 연결 재실패 - 중단");
                    emitter.completeWithError(e);
                    if (emitters.get(userId) != null && emitters.get(userId).contains(emitter)) {
                        removeEmitter(userId, emitter);
                    }
                    break;
                }
            }
        }
    }

    // 주기적으로 ping 전송(sse 연결 끊기지 않도록)
    @Scheduled(fixedRate = 15_000) // 15초마다
    public void sendPingToAll() {
        emitters.forEach((userId, deque) -> {
            Iterator<SseEmitter> iterator = deque.iterator();
            while (iterator.hasNext()) {
                SseEmitter emitter = iterator.next();
                try {
                    emitter.send(SseEmitter.event()
                            .name("ping")
                            .data("keep-alive"));
                } catch (IOException e) {
                    log.warn("❌ SSE 연결 실패 - userId={}, 원인={}", userId, e.getMessage());
                    emitter.completeWithError(e);
                    removeEmitter(userId, emitter);
                }
            }
        });
    }

    public void removeTargetEvent(Integer userId, AckRequestDto dto) {
        String dataJson = jsonUtil.toJson(dto.getData());
        SseEventWrapper target = new SseEventWrapper(dto.getEventName(), dataJson);
        Queue<SseEventWrapper> queue = retryQueue.get(userId);
        if (queue != null && !queue.isEmpty()) {
            boolean removed = queue.removeIf(e -> e.equals(target));
            if (removed) {
                log.debug("✅ 큐에서 이벤트 제거됨 - userId={}, eventName={}", userId, dto.getEventName());
            } else {
                log.debug("⚠️ 큐에 해당 이벤트 없음 - userId={}, eventName={}", userId, dto.getEventName());
            }
        }
    }

}
