package com.ssafy.hellojob.global.common.commitevent.listener;

import com.ssafy.hellojob.domain.interview.service.InterviewCompletionTracker;
import com.ssafy.hellojob.domain.sse.service.SSEService;
import com.ssafy.hellojob.global.common.commitevent.entity.InterviewFeedbackSavedEvent;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

import java.util.Map;

@Slf4j
@RequiredArgsConstructor
@Component
public class InterviewFeedbackSavedEventListener {

    private final InterviewCompletionTracker tracker;
    private final SSEService sseService;

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void handle(InterviewFeedbackSavedEvent event) {
        tracker.markFeedbackDone(event.getInterviewVideoId());

        if (tracker.tryMarkAndCheckAllDone(event.getInterviewVideoId(), true)) {
            log.debug("피드백 저장 함수에서 sse 호출함 !!!");
            sseService.sendToUser(event.getUserId(), "interview-feedback-completed", Map.of("interviewVideoId", event.getInterviewVideoId()));
        }
    }

}
