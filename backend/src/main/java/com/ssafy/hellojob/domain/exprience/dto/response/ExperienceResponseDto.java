package com.ssafy.hellojob.domain.exprience.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
public class ExperienceResponseDto {
    private Integer experienceId;
    private String experienceName;
    private String experienceDetail;
    private String experienceRole;
    private String experienceClient;
    private LocalDate experienceStartDate;
    private LocalDate experienceEndDate;
    private LocalDateTime updatedAt;

    @Builder
    public ExperienceResponseDto(Integer experienceId, String experienceName, String experienceDetail, String experienceRole, String experienceClient, LocalDate experienceStartDate, LocalDate experienceEndDate, LocalDateTime updatedAt) {
        this.experienceId = experienceId;
        this.experienceName = experienceName;
        this.experienceDetail = experienceDetail;
        this.experienceRole = experienceRole;
        this.experienceClient = experienceClient;
        this.experienceStartDate = experienceStartDate;
        this.experienceEndDate = experienceEndDate;
        this.updatedAt = updatedAt;
    }
}
