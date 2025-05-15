package com.ssafy.hellojob.domain.exprience.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class ExperienceCreateResponseDto {
    private Integer experienceId;
    private String message = "경험이 등록되었습니다.";

    @Builder
    public ExperienceCreateResponseDto(Integer experienceId) {
        this.experienceId = experienceId;
    }
}
