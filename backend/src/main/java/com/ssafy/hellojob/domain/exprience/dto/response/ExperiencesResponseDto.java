package com.ssafy.hellojob.domain.exprience.dto.response;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class ExperiencesResponseDto {
    private Integer experienceId;
    private String experienceName;
    private String experienceRole;
    private LocalDateTime updatedAt;
}
