package com.ssafy.hellojob.domain.jobroleanalysis.entity;

import com.ssafy.hellojob.domain.jobroleanalysis.dto.request.JobRoleAnalysisUpdateRequestDto;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.global.common.domain.BaseTimeEntity;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "job_role_analysis")
public class JobRoleAnalysis extends BaseTimeEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "job_role_analysis_id", nullable = false)
    private Integer jobRoleAnalysisId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;


    @Column(name = "company_id", nullable = false)
    private Integer companyId;

    @Column(name = "job_role_name", nullable = false, length = 100)
    private String jobRoleName;

    @Column(name = "job_role_title", nullable = false, length = 150)
    private String jobRoleTitle;

    @Column(name = "job_role_work", columnDefinition = "TEXT")
    private String jobRoleWork;

    @Column(name = "job_role_skills", columnDefinition = "TEXT")
    private String jobRoleSkills;

    @Column(name = "job_role_requirements", columnDefinition = "TEXT")
    private String jobRoleRequirements;

    @Column(name = "job_role_preferences", columnDefinition = "TEXT")
    private String jobRolePreferences;

    @Column(name = "job_role_etc", columnDefinition = "TEXT")
    private String jobRoleEtc;

    @Column(name = "job_role_view_count", nullable = false)
    private Integer jobRoleViewCount = 0;

    @Column(name = "public", nullable = false)
    private Boolean isPublic = true;

    @Enumerated(EnumType.STRING)
    @Column(name = "job_role_category")
    private JobRoleCategory jobRoleCategory;

    @Column(name = "job_role_bookmark_count", nullable = false)
    private Integer jobRoleBookmarkCount = 0;


    @Builder
    public JobRoleAnalysis(User user,
                           Integer companyId,
                           String jobRoleName,
                           String jobRoleTitle,
                           String jobRoleWork,
                           String jobRoleSkills,
                           String jobRoleRequirements,
                           String jobRolePreferences,
                           String jobRoleEtc,
                           Integer jobRoleViewCount,
                           Boolean isPublic,
                           JobRoleCategory jobRoleCategory,
                           Integer jobRoleBookmarkCount) {
        this.user = user;
        this.companyId = companyId;
        this.jobRoleName = jobRoleName;
        this.jobRoleTitle = jobRoleTitle;
        this.jobRoleWork = jobRoleWork;
        this.jobRoleSkills = jobRoleSkills;
        this.jobRoleRequirements = jobRoleRequirements;
        this.jobRolePreferences = jobRolePreferences;
        this.jobRoleEtc = jobRoleEtc;
        this.jobRoleViewCount = jobRoleViewCount;
        this.isPublic = isPublic;
        this.jobRoleCategory = jobRoleCategory;
        this.jobRoleBookmarkCount = jobRoleBookmarkCount;
    }


    public void setJobRoleBookmarkCount(Integer jobRoleBookmarkCount) {
        this.jobRoleBookmarkCount = jobRoleBookmarkCount;
    }

    public void setJobRoleViewCount(Integer jobRoleViewCount){
        this.jobRoleViewCount = jobRoleViewCount;
    }

    public void update(JobRoleAnalysisUpdateRequestDto requestDto) {
        this.jobRoleName = requestDto.getJobRoleName();
        this.jobRoleTitle = requestDto.getJobRoleTitle();
        this.jobRoleWork = requestDto.getJobRoleWork();
        this.jobRoleSkills = requestDto.getJobRoleSkills();
        this.jobRoleRequirements = requestDto.getJobRoleRequirements();
        this.jobRolePreferences = requestDto.getJobRolePreferences();
        this.jobRoleEtc = requestDto.getJobRoleEtc();
        this.jobRoleCategory = requestDto.getJobRoleCategory();
        // 필요한 필드만 수정
    }


}
