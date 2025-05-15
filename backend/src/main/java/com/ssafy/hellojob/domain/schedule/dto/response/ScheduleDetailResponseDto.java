package com.ssafy.hellojob.domain.schedule.dto.response;

import lombok.*;

import java.sql.Date;
import java.util.List;

@Getter
@Builder
@AllArgsConstructor
public class ScheduleDetailResponseDto {

    private Integer scheduleId;
    private Date scheduleStartDate;
    private Date scheduleEndDate;
    private String scheduleTitle;
    private String scheduleMemo;
    private String scheduleStatusName;
    private String scheduleStatusStep;

    private ScheduleDetailCoverLetter scheduleDetailCoverLetter;
    private ScheduleDetailCompanyAnalysis scheduleDetailCompanyAnalysis;
    private ScheduleDetailJobRoleSnapshot scheduleDetailJobRoleSnapshot;



}