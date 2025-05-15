package com.ssafy.hellojob.domain.schedule.dto.request;

import com.ssafy.hellojob.global.exception.ValidationMessage;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class ScheduleUpdateScheduleCoverLetterRequestDto {

    @NotNull(message = ValidationMessage.COVER_LETTER_ID_NOT_EMPTY)
    private Integer coverLetterId;

    @Builder
    public ScheduleUpdateScheduleCoverLetterRequestDto(Integer coverLetterId){
        this.coverLetterId= coverLetterId;
    }

}
