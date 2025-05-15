package com.ssafy.hellojob.domain.coverlettercontent.repository;

import com.ssafy.hellojob.domain.coverlettercontent.entity.CoverLetterExperience;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface CoverLetterExperienceRepository extends JpaRepository<CoverLetterExperience, Integer> {

    @Query("""
            SELECT cle.experienceId
            FROM CoverLetterExperience cle
            WHERE cle.coverLetterContent.contentId = :contentId
            AND cle.experienceId IS NOT NULL
            """)
    List<Integer> findExperiencesByContentId(
            @Param("contentId") Integer contentId);

    @Query("""
            SELECT cle.projectId
            FROM CoverLetterExperience cle
            WHERE cle.coverLetterContent.contentId = :contentId
            AND cle.projectId IS NOT NULL
            """)
    List<Integer> findProjectsByContentId(
            @Param("contentId") Integer contentId);

}

