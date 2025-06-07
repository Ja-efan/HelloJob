package com.ssafy.hellojob.global.common.commitevent.listener;

import com.ssafy.hellojob.domain.interview.entity.InterviewVideo;
import com.ssafy.hellojob.domain.interview.service.InterviewReadService;
import com.ssafy.hellojob.domain.interview.service.InterviewService;
import com.ssafy.hellojob.global.common.commitevent.entity.InterviewAnswerSavedEvent;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import org.springframework.transaction.event.TransactionPhase;
import org.springframework.transaction.event.TransactionalEventListener;

@Component
@Slf4j
@RequiredArgsConstructor
public class InterviewAnswerEventListener {

    private final InterviewReadService interviewReadService;
    private final InterviewService interviewService;

    @EventListener
    @TransactionalEventListener(phase = TransactionPhase.AFTER_COMMIT)
    public void onAnswerSaved(InterviewAnswerSavedEvent event) {
        log.info("✅ 커밋 완료 후 onAnswerSaved 로그: {}", event.getInterviewAnswer().getInterviewAnswer());

        InterviewVideo video = event.getInterviewAnswer().getInterviewVideo();
        Integer videoId = video.getInterviewVideoId();

        int totalQuestions = interviewReadService.countTotalQuestions(videoId); // 예: 5
        int savedAnswers = interviewReadService.countSavedAnswers(videoId); // null 아닌 답변 수

        if (totalQuestions == savedAnswers && !video.isFeedback()) {
            log.info("✅ 모든 답변 저장 완료. 자동으로 면접 종료 실행.");
            interviewService.endInterview(event.getUserId(), videoId);
        }
    }
}
