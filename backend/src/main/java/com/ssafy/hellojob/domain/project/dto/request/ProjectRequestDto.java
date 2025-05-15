package com.ssafy.hellojob.domain.project.dto.request;

import com.ssafy.hellojob.global.exception.ValidationMessage;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDate;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class ProjectRequestDto {

    @NotBlank(message = ValidationMessage.PROJECT_NAME_NOT_EMPTY)
    private String projectName;

    @NotBlank(message = ValidationMessage.PROJECT_INTRO_NOT_EMPTY)
    private String projectIntro;

    private String projectRole;
    private String projectSkills;
    private String projectDetail;
    private String projectClient;

    @NotNull(message = ValidationMessage.PROJECT_START_DATE_NOT_EMPTY)
    private LocalDate projectStartDate;

    @NotNull(message = ValidationMessage.PROJECT_END_DATE_NOT_EMPTY)
    private LocalDate projectEndDate;
}
