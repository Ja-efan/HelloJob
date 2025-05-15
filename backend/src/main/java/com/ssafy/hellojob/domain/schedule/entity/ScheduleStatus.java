package com.ssafy.hellojob.domain.schedule.entity;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "schedule_status")
public class ScheduleStatus {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "schedule_status_id", nullable = false)
    private Integer scheduleStatusId;

    @Column(name="schedule_status_name")
    private String scheduleStatusName;

    @Enumerated(EnumType.STRING)
    @Column(name="schedule_status_step")
    private ScheduleStatusStep scheduleStatusStep;
}
