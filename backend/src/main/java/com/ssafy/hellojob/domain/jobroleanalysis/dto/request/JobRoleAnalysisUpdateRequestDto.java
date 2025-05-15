package com.ssafy.hellojob.domain.jobroleanalysis.dto.request;

import com.ssafy.hellojob.domain.jobroleanalysis.entity.JobRoleCategory;
import com.ssafy.hellojob.global.exception.ValidationMessage;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@NoArgsConstructor
public class JobRoleAnalysisUpdateRequestDto {

    @NotNull(message = ValidationMessage.JOB_ROLE_ANALYSIS_JOB_ROLE_ANALYSIS_ID_NOT_EMPTY)
    private Integer jobRoleAnalysisId;

    @NotNull(message = ValidationMessage.JOB_ROLE_ANALYSIS_COMPANY_ID_NOT_EMPTY)
    private Integer companyId;

    @Size(max = 33, message = "직무명은 33자 이하로 입력해주세요.")
    @NotBlank(message = ValidationMessage.JOB_ROLE_ANALYSIS_JOB_ROLE_NAME_NOT_EMPTY)
    private String jobRoleName;

    @Size(max = 50, message = "직무 분석 제목은 50자 이하로 입력해주세요.")
    private String jobRoleTitle;

    private String jobRoleSkills;

    private String jobRoleWork;

    private String jobRoleRequirements;

    private String jobRolePreferences;

    private String jobRoleEtc;

    @NotNull(message = ValidationMessage.JOB_ROLE_ANALYSIS_JOB_ROLE_CATEGORY_NOT_EMPTY)
    private JobRoleCategory jobRoleCategory;

    private Boolean isPublic;
}
