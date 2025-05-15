package com.ssafy.hellojob.domain.schedule.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.sql.Date;

@Getter
@Setter
@NoArgsConstructor
public class ScheduleListResponseDto {

    private Integer scheduleId;
    private Date scheduleStartDate;
    private Date scheduleEndDate;
    private String scheduleTitle;
    private String scheduleStatusName;
    private String scheduleStatusStep;

    @Builder
    public ScheduleListResponseDto(Integer scheduleId, Date scheduleStartDate, Date scheduleEndDate, String scheduleTitle, String scheduleStatusName, String scheduleStatusStep){
        this.scheduleId = scheduleId;
        this.scheduleStartDate = scheduleStartDate;
        this.scheduleEndDate = scheduleEndDate;
        this.scheduleTitle = scheduleTitle;
        this.scheduleStatusName = scheduleStatusName;
        this.scheduleStatusStep = scheduleStatusStep;
    }

}