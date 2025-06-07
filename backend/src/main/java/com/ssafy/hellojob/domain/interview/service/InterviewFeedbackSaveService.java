package com.ssafy.hellojob.domain.interview.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.hellojob.domain.interview.dto.response.EndInterviewResponseDto;
import com.ssafy.hellojob.domain.interview.dto.response.InterviewFeedbackFastAPIResponseDto;
import com.ssafy.hellojob.domain.interview.dto.response.SingleInterviewFeedbackFastAPIResponseDto;
import com.ssafy.hellojob.domain.interview.entity.InterviewAnswer;
import com.ssafy.hellojob.domain.interview.entity.InterviewVideo;
import com.ssafy.hellojob.domain.interview.repository.InterviewAnswerRepository;
import com.ssafy.hellojob.domain.interview.repository.InterviewVideoRepository;
import com.ssafy.hellojob.global.common.commitevent.entity.InterviewFeedbackSavedEvent;
import com.ssafy.hellojob.global.exception.BaseException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.Map;

import static com.ssafy.hellojob.global.exception.ErrorCode.SERIALIZATION_FAIL;

@Slf4j
@Service
@RequiredArgsConstructor
public class InterviewFeedbackSaveService {

    private final InterviewAnswerRepository interviewAnswerRepository;
    private final InterviewVideoRepository interviewVideoRepository;
    private final InterviewReadService interviewReadService;
    private final ApplicationEventPublisher applicationEventPublisher;

    @Transactional
    public Map<String, String> saveTitle(Integer videoId, String title){
        InterviewVideo video = interviewReadService.findInterviewVideoByIdOrElseThrow(videoId);
        video.addTitle(title);
        return Map.of("message", "정상적으로 저장되었습니다.");
    }

    @Transactional
    public EndInterviewResponseDto saveFeedback(Integer userId, InterviewFeedbackFastAPIResponseDto fastAPIResponseDto, List<InterviewAnswer> interviewAnswers, InterviewVideo interviewVideo){
        
        log.debug("😎 saveFeedback 함수 들어옴");
        log.debug("😎 fastAPIResponseDto.getOverall_feedback() : {}", fastAPIResponseDto.getOverall_feedback());

        // 꼬리 질문 json 직렬화
        interviewVideoRepository.saveFeedback(interviewVideo.getInterviewVideoId(), fastAPIResponseDto.getOverall_feedback());

        for (SingleInterviewFeedbackFastAPIResponseDto singleInterviewFeedback : fastAPIResponseDto.getSingle_feedbacks()) {

            InterviewAnswer targetAnswer = interviewAnswers.stream()
                    .filter(ans -> ans.getInterviewAnswerId().equals(singleInterviewFeedback.getInterview_answer_id()))
                    .findFirst()
                    .orElseThrow(() -> new IllegalArgumentException("해당 interview_answer_id를 찾을 수 없습니다: " + singleInterviewFeedback.getInterview_answer_id()));

            String jsonFeedbacks;
            try {
                jsonFeedbacks = new ObjectMapper().writeValueAsString(singleInterviewFeedback.getFollow_up_questions());
            } catch (JsonProcessingException e) {
                log.debug("😱 삐상 !!!!!! interviewFeedback 저장 로직에서 json 파싱 에러 뜸 !!!!!: {}", e);
                throw new BaseException(SERIALIZATION_FAIL);
            }

            log.debug("jsonFeedbacks: {}", jsonFeedbacks);

            interviewAnswerRepository.saveInterviewFeedback(
                    singleInterviewFeedback.getInterview_answer_id(),
                    singleInterviewFeedback.getFeedback(),
                    jsonFeedbacks);
        }

        applicationEventPublisher.publishEvent(
                new InterviewFeedbackSavedEvent(userId, interviewVideo.getInterviewVideoId())
        );

        return EndInterviewResponseDto.builder()
                .interviewVideoId(interviewVideo.getInterviewVideoId())
                .build();
    }

}
