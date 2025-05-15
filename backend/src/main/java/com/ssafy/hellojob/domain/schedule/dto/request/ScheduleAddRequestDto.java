package com.ssafy.hellojob.domain.schedule.dto.request;

import com.ssafy.hellojob.global.exception.ValidationMessage;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.sql.Date;

@Getter
@Setter
@NoArgsConstructor
public class ScheduleAddRequestDto {

    private Date scheduleStartDate;

    private Date scheduleEndDate;

    @Size(max = 33, message = "스케줄 제목은 33자 이하로 입력해주세요.")
    @NotBlank(message = ValidationMessage.SCHEDULE_TITLE_NOT_EMPTY)
    private String scheduleTitle;

    @Size(max = 166, message = "스케줄 메모는 166자 이하로 입력해주세요.")
    private String scheduleMemo;

    @NotBlank(message = ValidationMessage.SCHEDULE_STATUS_NOT_EMPTY)
    private String scheduleStatusName;

    private Integer coverLetterId;

    @Builder
    private ScheduleAddRequestDto(Date scheduleStartDate, Date scheduleEndDate, String scheduleTitle, String scheduleMemo, String scheduleStatusName, Integer coverLetterId){
        this.scheduleStartDate=scheduleStartDate;
        this.scheduleEndDate=scheduleEndDate;
        this.scheduleTitle=scheduleTitle;
        this.scheduleMemo=scheduleMemo;
        this.scheduleStatusName=scheduleStatusName;
        this.coverLetterId=coverLetterId;
    }

}
