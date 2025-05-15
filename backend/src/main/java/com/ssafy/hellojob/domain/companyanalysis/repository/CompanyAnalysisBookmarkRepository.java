package com.ssafy.hellojob.domain.companyanalysis.repository;

import com.ssafy.hellojob.domain.companyanalysis.entity.CompanyAnalysis;
import com.ssafy.hellojob.domain.companyanalysis.entity.CompanyAnalysisBookmark;
import com.ssafy.hellojob.domain.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface CompanyAnalysisBookmarkRepository extends JpaRepository<CompanyAnalysisBookmark, Integer> {

    boolean existsByUser_UserIdAndCompanyAnalysis_CompanyAnalysisId(Integer userId, Integer companyAnalysisId);
    Optional<CompanyAnalysisBookmark> findByUserAndCompanyAnalysis(User user, CompanyAnalysis companyAnalysis);

    List<CompanyAnalysisBookmark> findAllByUser(User user);

    List<CompanyAnalysisBookmark> findAllByUserAndCompanyAnalysis_Company_CompanyId(User user, Integer companyId);



}
