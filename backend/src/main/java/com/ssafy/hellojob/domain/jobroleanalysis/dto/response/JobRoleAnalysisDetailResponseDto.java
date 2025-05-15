package com.ssafy.hellojob.domain.jobroleanalysis.dto.response;

import com.ssafy.hellojob.domain.jobroleanalysis.entity.JobRoleCategory;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;

@Getter
@Builder
@AllArgsConstructor
public class JobRoleAnalysisDetailResponseDto {

    private Integer jobRoleAnalysisId;        // 직무 분석 레포트 아이디
    private String companyName;             // 기업명
    private String jobRoleName;             // 직무명
    private String jobRoleAnalysisTitle;
    private String jobRoleWork;
    private String jobRoleSkills;           // 기술스택
    private String jobRoleRequirements;     // 자격요건
    private String jobRolePreferences;      // 우대사항
    private String jobRoleEtc;               // 기타 입력사항
    private Integer jobRoleViewCount;       // 조회수
    private Boolean isPublic;               // 공개 여부 (true/false)
    private JobRoleCategory jobRoleCategory; // 직무 카테고리 (Enum 타입 그대로)
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;         // 최종 수정날짜
    private Integer jobRoleAnalysisBookmarkCount; // 총 즐겨찾기 수
    private Boolean bookmark;                // 현재 유저가 즐겨찾기 했는지 여부

}

