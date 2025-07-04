package com.ssafy.hellojob.domain.project.entity;

import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterExperience;
import com.ssafy.hellojob.domain.project.dto.request.ProjectRequestDto;
import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.global.common.domain.BaseTimeEntity;
import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@Table(name = "project")
public class Project extends BaseTimeEntity {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "project_id")
    private Integer projectId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(name = "project_name", nullable = false, length = 100)
    private String projectName;

    @Column(name = "project_intro", nullable = false)
    private String projectIntro;

    @Column(name = "project_role", length = 50)
    private String projectRole;

    @Column(name = "project_skills", columnDefinition = "TEXT")
    private String projectSkills;

    @Column(name = "project_detail", columnDefinition = "TEXT")
    private String projectDetail;

    @Column(name = "project_client", length = 100)
    private String projectClient;

    @Column(name = "project_start_date")
    private LocalDate projectStartDate;

    @Column(name = "project_end_date")
    private LocalDate projectEndDate;

    @OneToMany(mappedBy = "project", cascade = CascadeType.REMOVE, orphanRemoval = true)
    private List<CoverLetterExperience> coverLetterExperiences = new ArrayList<>();

    @Builder
    public Project(User user,
                   String projectName,
                   String projectIntro,
                   String projectRole,
                   String projectSkills,
                   String projectDetail,
                   String projectClient,
                   LocalDate projectStartDate,
                   LocalDate projectEndDate) {
        this.user = user;
        this.projectName = projectName;
        this.projectIntro = projectIntro;
        this.projectRole = projectRole;
        this.projectSkills = projectSkills;
        this.projectDetail = projectDetail;
        this.projectClient = projectClient;
        this.projectStartDate = projectStartDate;
        this.projectEndDate = projectEndDate;
    }

    public void updateProject(ProjectRequestDto project) {
        this.projectName = project.getProjectName();
        this.projectIntro = project.getProjectIntro();
        this.projectRole = project.getProjectRole();
        this.projectSkills = project.getProjectSkills();
        this.projectDetail = project.getProjectDetail();
        this.projectClient = project.getProjectClient();
        this.projectStartDate = project.getProjectStartDate();
        this.projectEndDate = project.getProjectEndDate();
    }
}
