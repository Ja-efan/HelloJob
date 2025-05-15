package com.ssafy.hellojob.domain.companyanalysis.entity;

import com.ssafy.hellojob.domain.jobroleanalysis.entity.JobRoleAnalysis;
import com.ssafy.hellojob.domain.user.entity.User;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "company_analysis_bookmark")
public class CompanyAnalysisBookmark {


    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "company_analysis_bookmark_id", nullable = false)
    private Integer companyAnalysisBookmarkId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "company_analysis_id", nullable = false)
    private CompanyAnalysis companyAnalysis;

    @Builder
    public CompanyAnalysisBookmark(User user, CompanyAnalysis companyAnalysis){
        this.user = user;
        this.companyAnalysis = companyAnalysis;
    }

}
