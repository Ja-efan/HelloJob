package com.ssafy.hellojob.domain.jobrolesnapshot.entity;

import com.ssafy.hellojob.global.common.domain.BaseTimeEntity;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "job_role_snapshot")
public class JobRoleSnapshot extends BaseTimeEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "job_role_snapshot_id")
    private Integer jobRoleSnapshotId;

    @Column(name = "job_role_analysis_id")
    private Integer jobRoleAnalysisId;

    @Column(name = "company_name", nullable = false, length = 100)
    private String companyName;

    @Column(name = "job_role_snapshot_name", nullable = false, length = 100)
    private String jobRoleSnapshotName;

    @Column(name = "job_role_snapshot_title", length = 150)
    private String jobRoleSnapshotTitle;

    @Column(name = "job_role_snapshot_work", columnDefinition = "TEXT")
    private String jobRoleSnapshotWork;

    @Column(name = "job_role_snapshot_skills", columnDefinition = "TEXT")
    private String jobRoleSnapshotSkills;

    @Column(name = "job_role_snapshot_requirements", columnDefinition = "TEXT")
    private String jobRoleSnapshotRequirements;

    @Column(name = "job_role_snapshot_preferences", columnDefinition = "TEXT")
    private String jobRoleSnapshotPreferences;

    @Column(name = "job_role_snapshot_etc", columnDefinition = "TEXT")
    private String jobRoleSnapshotEtc;

    @Column(name = "job_role_snapshot_category", length = 50)
    private String jobRoleSnapshotCategory;

    @Builder
    public JobRoleSnapshot(Integer jobRoleSnapshotId, Integer jobRoleAnalysisId, String companyName, String jobRoleSnapshotName, String jobRoleSnapshotTitle, String jobRoleSnapshotWork, String jobRoleSnapshotSkills, String jobRoleSnapshotRequirements, String jobRoleSnapshotPreferences, String jobRoleSnapshotEtc, String jobRoleSnapshotCategory) {
        this.jobRoleSnapshotId = jobRoleSnapshotId;
        this.jobRoleAnalysisId = jobRoleAnalysisId;
        this.companyName = companyName;
        this.jobRoleSnapshotName = jobRoleSnapshotName;
        this.jobRoleSnapshotTitle = jobRoleSnapshotTitle;
        this.jobRoleSnapshotWork = jobRoleSnapshotWork;
        this.jobRoleSnapshotSkills = jobRoleSnapshotSkills;
        this.jobRoleSnapshotRequirements = jobRoleSnapshotRequirements;
        this.jobRoleSnapshotPreferences = jobRoleSnapshotPreferences;
        this.jobRoleSnapshotEtc = jobRoleSnapshotEtc;
        this.jobRoleSnapshotCategory = jobRoleSnapshotCategory;
    }
}
