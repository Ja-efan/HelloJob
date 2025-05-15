package com.ssafy.hellojob.domain.project.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class ProjectResponseDto {
    private Integer projectId;
    private String projectName;
    private String projectIntro;
    private String projectRole;
    private String projectSkills;
    private String projectDetail;
    private String projectClient;
    private LocalDate projectStartDate;
    private LocalDate projectEndDate;
    private LocalDateTime updatedAt;
}
