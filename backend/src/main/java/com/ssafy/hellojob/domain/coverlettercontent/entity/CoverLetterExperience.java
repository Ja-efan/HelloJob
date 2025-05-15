package com.ssafy.hellojob.domain.coverlettercontent.entity;

import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterContent;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "cover_letter_experience")
public class CoverLetterExperience {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "cover_letter_experience")
    private Integer coverLetterExperienceId;

    @Column(name = "experience_id")
    private Integer experienceId;

    @Column(name = "project_id")
    private Integer projectId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "cover_letter_content_id", nullable = false)
    private CoverLetterContent coverLetterContent;

    @Builder
    public CoverLetterExperience(Integer coverLetterExperienceId, Integer experienceId, Integer projectId, CoverLetterContent coverLetterContent) {
        this.coverLetterExperienceId = coverLetterExperienceId;
        this.experienceId = experienceId;
        this.projectId = projectId;
        this.coverLetterContent = coverLetterContent;
    }
}
