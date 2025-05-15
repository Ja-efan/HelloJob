package com.ssafy.hellojob.domain.jobroleanalysis.entity;

import com.ssafy.hellojob.domain.user.entity.User;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "job_role_analysis_bookmark")
public class JobRoleAnalysisBookmark {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "job_role_analysis_bookmark_id", nullable = false)
    private Integer jobRoleAnalysisBookmarkId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "job_role_analysis_id", nullable = false)
    private JobRoleAnalysis jobRoleAnalysis;

    @Builder
    public JobRoleAnalysisBookmark(User user, JobRoleAnalysis jobRoleAnalysis){
        this.user = user;
        this.jobRoleAnalysis = jobRoleAnalysis;
    }

}
