package com.ssafy.hellojob.domain.schedule.entity;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public enum ScheduleStatusStep {

    PENDING,
    IN_PROGRESS,
    DONE

}
