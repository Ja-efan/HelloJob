package com.ssafy.hellojob.domain.companyanalysis.repository;

import com.ssafy.hellojob.domain.companyanalysis.entity.CompanyAnalysis;
import com.ssafy.hellojob.domain.companyanalysis.entity.NewsAnalysis;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface NewsAnalysisRepository extends JpaRepository<NewsAnalysis, Integer> {

}
