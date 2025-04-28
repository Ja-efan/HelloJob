package com.ssafy.hellojob.domain.project.repository;

import com.ssafy.hellojob.domain.project.dto.response.ProjectsResponseDto;
import com.ssafy.hellojob.domain.project.entity.Project;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface ProjectRepository extends JpaRepository<Project, Integer> {
    Optional<Project> findById(Integer id);

    @Query("SELECT new com.ssafy.hellojob.domain.project.dto.response.ProjectsResponseDto("+
    "p.projectId, p.projectName, p.projectStartDate, p.projectEndDate) " +
    "FROM Project p WHERE p.user.userId = :userId")
    List<ProjectsResponseDto> findByUserId(@Param("userId") Integer userId);
}
