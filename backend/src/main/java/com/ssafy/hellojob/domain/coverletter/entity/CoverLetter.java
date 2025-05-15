package com.ssafy.hellojob.domain.coverletter.entity;

import com.ssafy.hellojob.domain.companyanalysis.entity.CompanyAnalysis;
import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContent;
import com.ssafy.hellojob.domain.jobrolesnapshot.entity.JobRoleSnapshot;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.global.common.domain.BaseTimeEntity;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.hibernate.annotations.ColumnDefault;

import java.util.ArrayList;
import java.util.List;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "cover_letter")
public class CoverLetter extends BaseTimeEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "cover_letter_id")
    private Integer coverLetterId;

    @ManyToOne
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "cover_letter_title", length = 90, nullable = false)
    private String coverLetterTitle;

    @ManyToOne
    @JoinColumn(name = "company_analysis_id", nullable = false)
    private CompanyAnalysis companyAnalysis;

    @OneToOne(cascade = CascadeType.ALL, orphanRemoval = true)
    @JoinColumn(name = "job_role_analysis_snapshot")
    private JobRoleSnapshot jobRoleSnapshot;

    @Column(name = "finish", nullable = false)
    @ColumnDefault("false")
    private boolean finish = false;

    @OneToMany(mappedBy = "coverLetter", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<CoverLetterContent> contents = new ArrayList<>();

    @Builder
    public CoverLetter(Integer coverLetterId, User user, String coverLetterTitle, CompanyAnalysis companyAnalysis, JobRoleSnapshot jobRoleSnapshot, boolean finish) {
        this.coverLetterId = coverLetterId;
        this.user = user;
        this.coverLetterTitle = coverLetterTitle;
        this.companyAnalysis = companyAnalysis;
        this.jobRoleSnapshot = jobRoleSnapshot;
        this.finish = finish;
    }

    public void updateFinish(boolean finish) {
        this.finish = finish;
    }
}
