package com.ssafy.hellojob.domain.schedule.dto.response;

import jakarta.persistence.Column;
import lombok.*;

@Getter
@Builder
@AllArgsConstructor
public class ScheduleDetailJobRoleSnapshot {
    private Integer jobRoleSnapshotId;
    private Integer jobRoleAnalysisId;
    private String jobRoleSnapshotName;
    private String jobRoleSnapshotTitle;
    private String jobRoleSnapshotWork;
    private String jobRoleSnapshotSkills;
    private String jobRoleSnapshotRequirements;
    private String jobRoleSnapshotPreferences;
    private String jobRoleSnapshotEtc;
    private String jobRoleSnapshotCategory;

}

