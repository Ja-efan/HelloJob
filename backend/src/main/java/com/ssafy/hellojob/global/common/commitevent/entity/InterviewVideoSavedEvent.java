package com.ssafy.hellojob.global.common.commitevent.entity;

import lombok.Getter;

@Getter
public class InterviewVideoSavedEvent {

    private Integer userId;
    private Integer interviewVideoId;

    public InterviewVideoSavedEvent(Integer userId, Integer interviewVideoId) {
        this.userId = userId;
        this.interviewVideoId = interviewVideoId;
    }
}
