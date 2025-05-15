package com.ssafy.hellojob.domain.jobroleanalysis.repository;

import com.ssafy.hellojob.domain.jobroleanalysis.entity.JobRoleAnalysis;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface JobRoleAnalysisRepository extends JpaRepository<JobRoleAnalysis, Integer> {

    @Query("SELECT j.user.userId FROM JobRoleAnalysis j WHERE j.jobRoleAnalysisId = :jobRoleAnalysisId")
    Optional<Integer> findUserIdByJobRoleAnalysisId(@Param("jobRoleAnalysisId") Integer jobRoleAnalysisId);

}
