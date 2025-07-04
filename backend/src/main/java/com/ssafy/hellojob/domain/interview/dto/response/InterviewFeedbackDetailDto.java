package com.ssafy.hellojob.domain.interview.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class InterviewFeedbackDetailDto {

    private Integer interviewAnswerId;
    private String interviewQuestion;
    private String interviewAnswer;
    private String interviewAnswerFeedback;
    private List<String> interviewAnswerFollowUpQuestion;
    private String interviewAnswerVideoUrl;
    private String interviewAnswerLength;
}

