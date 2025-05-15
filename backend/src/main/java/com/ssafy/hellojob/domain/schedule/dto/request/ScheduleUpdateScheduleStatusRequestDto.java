package com.ssafy.hellojob.domain.schedule.dto.request;

import com.ssafy.hellojob.global.exception.ValidationMessage;
import jakarta.validation.constraints.NotBlank;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class ScheduleUpdateScheduleStatusRequestDto {

    @NotBlank(message = ValidationMessage.SCHEDULE_STATUS_NOT_EMPTY)
    private String scheduleStatusName;

    @Builder
    public ScheduleUpdateScheduleStatusRequestDto(String scheduleStatusName){
        this.scheduleStatusName= scheduleStatusName;
    }

}
