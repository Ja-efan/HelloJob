package com.ssafy.hellojob.global.common.commitevent.entity;

import com.ssafy.hellojob.domain.interview.entity.InterviewAnswer;

public class InterviewAnswerSavedEvent {
    private final InterviewAnswer interviewAnswer;
    private Integer userId;

    public InterviewAnswerSavedEvent(InterviewAnswer interviewAnswer, Integer userId) {
        this.interviewAnswer = interviewAnswer;
        this.userId = userId;
    }

    public InterviewAnswer getInterviewAnswer() {
        return interviewAnswer;
    }

    public Integer getUserId() {
        return userId;
    }
}
