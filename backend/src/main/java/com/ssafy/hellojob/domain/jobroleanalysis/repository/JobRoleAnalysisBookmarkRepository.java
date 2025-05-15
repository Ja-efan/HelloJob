package com.ssafy.hellojob.domain.jobroleanalysis.repository;

import com.ssafy.hellojob.domain.jobroleanalysis.entity.JobRoleAnalysis;
import com.ssafy.hellojob.domain.jobroleanalysis.entity.JobRoleAnalysisBookmark;
import com.ssafy.hellojob.domain.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface JobRoleAnalysisBookmarkRepository extends JpaRepository<JobRoleAnalysisBookmark, Integer> {

    boolean existsByUserAndJobRoleAnalysis(User user, JobRoleAnalysis jobRoleAnalysis);

    Optional<JobRoleAnalysisBookmark> findByUserAndJobRoleAnalysis(User user, JobRoleAnalysis jobRoleAnalysis);

    List<JobRoleAnalysisBookmark> findAllByUser(User user);

    List<JobRoleAnalysisBookmark> findByUserAndJobRoleAnalysis_CompanyId(User user, Integer companyId);

}
