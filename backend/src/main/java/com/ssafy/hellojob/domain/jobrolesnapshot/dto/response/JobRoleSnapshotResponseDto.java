package com.ssafy.hellojob.domain.jobrolesnapshot.dto.response;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
public class JobRoleSnapshotResponseDto {
    private Integer JobRoleSnapshotId;
    private String companyName;
    private String jobRoleSnapshotName;
    private String jobRoleSnapshotTitle;
    private String jobRoleSnapshotSkills;
    private String jobRoleSnapshotWork;
    private String jobRoleSnapshotRequirements;
    private String jobRoleSnapshotPreferences;
    private String jobRoleSnapshotEtc;
    private String jobRoleSnapshotCategory;
}
