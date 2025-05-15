package com.ssafy.hellojob.domain.project.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
public class ProjectsResponseDto {
    private Integer projectId;
    private String projectName;
    private String projectIntro;
    private String projectSkills;
    private LocalDateTime updatedAt;

    @Builder
    public ProjectsResponseDto(Integer projectId, String projectName, String projectIntro, String projectSkills, LocalDateTime updatedAt) {
        this.projectId = projectId;
        this.projectName = projectName;
        this.projectIntro = projectIntro;
        this.projectSkills = projectSkills;
        this.updatedAt = updatedAt;
    }
}
