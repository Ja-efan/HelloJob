package com.ssafy.hellojob.domain.exprience.repository;

import com.ssafy.hellojob.domain.exprience.dto.response.ExperiencesResponseDto;
import com.ssafy.hellojob.domain.exprience.entity.Experience;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;
import java.util.Optional;

public interface ExperienceRepository extends JpaRepository<Experience, Integer> {

    Optional<Experience> findByExperienceId(Integer experienceId);

    @Query("SELECT new com.ssafy.hellojob.domain.exprience.dto.response.ExperiencesResponseDto( " +
    "e.experienceId, e.experienceName, e.experienceRole, e.updatedAt) " +
    "FROM Experience e WHERE e.user.userId = :userId " +
    "ORDER BY e.updatedAt DESC")
    List<ExperiencesResponseDto> findExperiencesByUserId(@Param("userId") Integer userId);

    @Query("SELECT new com.ssafy.hellojob.domain.exprience.dto.response.ExperiencesResponseDto( " +
            "e.experienceId, e.experienceName, e.experienceRole, e.updatedAt) " +
            "FROM Experience e WHERE e.user.userId = :userId " +
            "ORDER BY e.updatedAt DESC")
    Page<ExperiencesResponseDto> findExperiencesPageByUserId(@Param("userId") Integer userId, Pageable pageable);
}
