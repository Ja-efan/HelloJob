package com.ssafy.hellojob.domain.schedule.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class ScheduleIdResponseDto {

    private Integer scheduleId;

    @Builder
    public ScheduleIdResponseDto(Integer scheduleId){
        this.scheduleId = scheduleId;
    }

}
