package com.ssafy.hellojob.domain.interview.entity;

import com.ssafy.hellojob.domain.interview.repository.InterviewVideoRepository;
import com.ssafy.hellojob.global.exception.BaseException;
import com.ssafy.hellojob.global.exception.ErrorCode;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Service;

@Service
@Slf4j
@RequiredArgsConstructor
public class InterviewVideoReadService {

    private final InterviewVideoRepository interviewVideoRepository;

    public InterviewVideo findById(Integer interviewVideoId){
        return interviewVideoRepository.findById(interviewVideoId)
                .orElseThrow(() -> new BaseException(ErrorCode.INTERVIEW_VIDEO_NOT_FOUND));
    }

}
