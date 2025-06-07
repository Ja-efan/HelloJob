package com.ssafy.hellojob.global.common.commitevent.entity;

import lombok.Getter;

@Getter
public class InterviewFeedbackSavedEvent {

    private Integer userId;
    private Integer interviewVideoId;

    public InterviewFeedbackSavedEvent(Integer userId, Integer interviewVideoId){
        this.userId = userId;
        this.interviewVideoId = interviewVideoId;
    }

}
