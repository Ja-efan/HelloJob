package com.ssafy.hellojob.domain.exprience.dto.request;

import com.ssafy.hellojob.global.exception.ValidationMessage;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class ExperienceRequestDto {

    @NotBlank(message = ValidationMessage.EXPERIENCE_NAME_NOT_EMPTY)
    private String experienceName;

    @NotBlank(message = ValidationMessage.EXPERIENCE_DETAIL_NOT_EMPTY)
    private String experienceDetail;

    private String experienceRole;
    private String experienceClient;

    @NotNull(message = ValidationMessage.EXPERIENCE_START_DATE_NOT_EMPTY)
    private LocalDate experienceStartDate;

    @NotNull(message = ValidationMessage.EXPERIENCE_END_DATE_NOT_EMPTY)
    private LocalDate experienceEndDate;
}
