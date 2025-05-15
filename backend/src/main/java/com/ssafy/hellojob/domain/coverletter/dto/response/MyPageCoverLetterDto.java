package com.ssafy.hellojob.domain.coverletter.dto.response;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class MyPageCoverLetterDto {
    private Integer coverLetterId;
    private String coverLetterTitle;
    private String companyName;
    private String jobRoleName;
    private String jobRoleCategory;
    private boolean finish;
    private LocalDateTime updatedAt;
}
