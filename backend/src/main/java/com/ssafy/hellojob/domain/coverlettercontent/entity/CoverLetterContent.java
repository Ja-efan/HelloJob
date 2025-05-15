package com.ssafy.hellojob.domain.coverlettercontent.entity;

import com.ssafy.hellojob.domain.coverlettercontent.dto.request.CoverLetterUpdateRequestDto;
import com.ssafy.hellojob.domain.coverletter.entity.CoverLetter;
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
@Table(name = "cover_letter_content")
public class CoverLetterContent extends BaseTimeEntity {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "cover_letter_content_id")
    private Integer contentId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "cover_letter_id", nullable = false)
    private CoverLetter coverLetter;

    @Column(name = "cover_letter_content_number", nullable = false)
    private Integer contentNumber;

    @Column(name = "cover_letter_content_question", nullable = false, columnDefinition = "TEXT")
    private String contentQuestion;

    @Column(name = "cover_letter_content_detail", columnDefinition = "TEXT")
    private String contentDetail;

    @Column(name = "cover_letter_content_length")
    private Integer contentLength;

    @Enumerated(EnumType.STRING)
    @ColumnDefault("'PENDING'")
    @Column(name = "cover_letter_content_status", nullable = false)
    private CoverLetterContentStatus contentStatus = CoverLetterContentStatus.PENDING;

    @Column(name = "cover_letter_content_first_prompt", length = 4500)
    private String contentFirstPrompt;

    @OneToOne(mappedBy = "coverLetterContent", cascade = CascadeType.ALL, orphanRemoval = true)
    private ChatLog chatLog;

    @OneToMany(mappedBy = "coverLetterContent", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<CoverLetterExperience> experiences = new ArrayList<>();

    @Builder
    public CoverLetterContent(Integer contentId, CoverLetter coverLetter, Integer contentNumber, String contentQuestion, String contentDetail, Integer contentLength, CoverLetterContentStatus contentStatus, String contentFirstPrompt) {
        this.contentId = contentId;
        this.coverLetter = coverLetter;
        this.contentNumber = contentNumber;
        this.contentQuestion = contentQuestion;
        this.contentDetail = contentDetail;
        this.contentLength = contentLength;
        this.contentStatus = contentStatus;
        this.contentFirstPrompt = contentFirstPrompt;
    }

    public void updateCoverLetterContent(CoverLetterUpdateRequestDto requestDto) {
        this.contentDetail = requestDto.getContentDetail();
        if (requestDto.getContentStatus() != null) {
            this.contentStatus = requestDto.getContentStatus();
        }
    }

    public void updateCoverLetterContentWithChat(String contentDetail) {
        this.contentDetail = contentDetail;
    }

    public void updateContentStatus(CoverLetterContentStatus status) {
        this.contentStatus = status;
    }
}
