package com.ssafy.hellojob.global.common.commitevent.listener;

import com.ssafy.hellojob.domain.interview.service.InterviewCompletionTracker;
import com.ssafy.hellojob.domain.sse.service.SSEService;
import com.ssafy.hellojob.global.common.commitevent.entity.InterviewVideoSavedEvent;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

import java.util.Map;

@RequiredArgsConstructor
@Component
public class InterviewVideoSavedEventListener {

    private final InterviewCompletionTracker tracker;
    private final SSEService sseService;

    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void handle(InterviewVideoSavedEvent event) {
        tracker.markVideoDone(event.getInterviewVideoId());

        if (tracker.isAllDone(event.getInterviewVideoId())) {
            sseService.sendToUser(event.getUserId(), "interview-feedback-completed", Map.of("interviewVideoId", event.getInterviewVideoId()));
        }
    }
}
