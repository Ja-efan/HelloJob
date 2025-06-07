package com.ssafy.hellojob.domain.interview.service;

import com.ssafy.hellojob.domain.interview.entity.InterviewAnswer;
import com.ssafy.hellojob.domain.interview.repository.InterviewAnswerRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionSynchronizationManager;

@Slf4j
@Service
@RequiredArgsConstructor
public class InterviewAnswerContentSaveService {

    private final InterviewAnswerRepository interviewAnswerRepository;

    // 저장 함수
    @Transactional(propagation = Propagation.REQUIRES_NEW)
    public void saveAnswer(String answer, InterviewAnswer interviewAnswer){
        log.debug("😎 id: {}, saveAnswer에 들어온 값: {}", interviewAnswer.getInterviewAnswerId(), answer);

        if(answer == null){
            answer = "stt 변환에 실패했습니다";
        }

        try{
            log.debug("🔍 트랜잭션 활성 여부: {}", TransactionSynchronizationManager.isActualTransactionActive());

            interviewAnswerRepository.saveInterviewAnswer(interviewAnswer.getInterviewAnswerId(), answer);
            interviewAnswerRepository.flush();

            log.debug("✅ flush 완료");


            log.debug("😎 id: {} 답변 저장 완", interviewAnswer.getInterviewAnswerId());
        } catch(Exception e){
            log.debug("😱 id: {} 삐상 !!!!!!!!!!! 답변 db에 저장 중 에러 발생 !!!!!!!!!!!!!!!!!!!!!!!", interviewAnswer.getInterviewAnswerId());
            log.debug("{}", e);
        }

        InterviewAnswer answer1 = interviewAnswerRepository.findById(interviewAnswer.getInterviewAnswerId())
                .orElseThrow();
        log.debug("💾 저장 직후 DB에서 해당 ID 조회: {}", answer1.getInterviewAnswer());

    }

    @Transactional
    public void saveAllAnswerData(String url, String videoLength, InterviewAnswer interviewAnswer) {
//        interviewAnswer.addInterviewVideoUrl(url);
//        interviewAnswer.addVideoLength(videoLength);
//        interviewAnswerRepository.save(interviewAnswer);
        interviewAnswerRepository.flush();

        log.debug("😎 S3 url db에 저장 완.");
    }


}
