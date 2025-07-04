package com.ssafy.hellojob.domain.interview.entity;

import com.ssafy.hellojob.domain.coverletter.entity.CoverLetter;
import com.ssafy.hellojob.domain.user.entity.User;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "cover_letter_interview")
public class CoverLetterInterview {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "cover_letter_interview_id", nullable = false)
    private Integer coverLetterInterviewId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "cover_letter_id")
    private CoverLetter coverLetter;

    public static CoverLetterInterview of(User user, CoverLetter coverLetter){
        CoverLetterInterview interview = new CoverLetterInterview();
        interview.user = user;
        interview.coverLetter = coverLetter;
        return interview;
    }

}
