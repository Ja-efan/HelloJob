package com.ssafy.hellojob.domain.exprience.entity;

import com.ssafy.hellojob.domain.exprience.dto.request.ExperienceRequestDto;
import com.ssafy.hellojob.domain.exprience.dto.response.ExperienceResponseDto;
import com.ssafy.hellojob.domain.user.entity.User;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "experience")
public class Experience {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "experience_id")
    private Integer experienceId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "experience_name", nullable = false, length = 100)
    private String experienceName;

    @Column(name = "experience_detail", nullable = false, columnDefinition = "TEXT")
    private String experienceDetail;

    @Column(name = "experience_role", length = 50)
    private String experienceRole;

    @Column(name = "experience_client", length = 100)
    private String experienceClient;

    @Column(name = "experience_start_date", nullable = false)
    private LocalDate experienceStartDate;

    @Column(name = "experience_end_date", nullable = false)
    private LocalDate experienceEndDate;

    @Column(name = "created_at", nullable = false)
    private LocalDateTime createdAt;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    @Builder
    public Experience(Integer experienceId, User user, String experienceName, String experienceDetail, String experienceRole, String experienceClient, LocalDate experienceStartDate, LocalDate experienceEndDate) {
        this.experienceId = experienceId;
        this.user = user;
        this.experienceName = experienceName;
        this.experienceDetail = experienceDetail;
        this.experienceRole = experienceRole;
        this.experienceClient = experienceClient;
        this.experienceStartDate = experienceStartDate;
        this.experienceEndDate = experienceEndDate;
    }

    public void updateExperience(ExperienceRequestDto experience) {
        this.experienceName = experience.getExperienceName();
        this.experienceRole = experience.getExperienceRole();
        this.experienceDetail = experience.getExperienceDetail();
        this.experienceClient = experience.getExperienceClient();
        this.experienceStartDate = experience.getExperienceStartDate();
        this.experienceEndDate = experience.getExperienceEndDate();
    }

    @PrePersist
    public void onCreate() {
        LocalDateTime now = LocalDateTime.now();
        this.createdAt = now;
        this.updatedAt = now;
    }

    @PreUpdate
    public void onUpdate() {
        this.updatedAt = LocalDateTime.now();
    }
}
