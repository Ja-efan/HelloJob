package com.ssafy.hellojob.domain.companyanalysis.repository;

import com.ssafy.hellojob.domain.companyanalysis.entity.CompanyAnalysis;
import com.ssafy.hellojob.domain.companyanalysis.entity.DartAnalysis;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface DartAnalysisRepository extends JpaRepository<DartAnalysis, Integer> {

}
