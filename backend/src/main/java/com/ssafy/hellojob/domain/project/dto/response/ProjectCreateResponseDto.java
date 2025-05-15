package com.ssafy.hellojob.domain.project.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class ProjectCreateResponseDto {
    private Integer projectId;
    private String message = "프로젝트가 등록되었습니다.";

    @Builder
    public ProjectCreateResponseDto(Integer projectId) {
        this.projectId = projectId;
    }
}
