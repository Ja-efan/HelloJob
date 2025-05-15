package com.ssafy.hellojob.domain.company.repository;

import com.ssafy.hellojob.domain.company.dto.CompanyListDto;
import com.ssafy.hellojob.domain.company.entity.Company;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface CompanyRepository extends JpaRepository<Company, Integer> {

    @Query("SELECT new com.ssafy.hellojob.domain.company.dto.CompanyListDto(c.companyId, c.companyName, c.companyLocation, c.companySize, c.companyIndustry, c.updatedAt) " +
            "FROM Company c WHERE c.companyVisible IS TRUE ORDER BY c.updatedAt DESC")
    List<CompanyListDto> getAllCompany();

    @Query("SELECT new com.ssafy.hellojob.domain.company.dto.CompanyListDto(c.companyId, c.companyName, c.companyLocation, c.companySize, c.companyIndustry, c.updatedAt) " +
            "FROM Company c WHERE c.companyName LIKE CONCAT(:companyName, '%') AND c.companyVisible IS TRUE ORDER BY c.updatedAt DESC")
    List<CompanyListDto> getCompanyByCompanyName(@Param("companyName") String companyName);

    @Query("SELECT c.companyName FROM Company c WHERE c.companyId = :companyId")
    Optional<String> getCompanyNameByCompanyId(@Param("companyId") Integer companyId);

    Optional<Company> findByCompanyId(@Param("companyId") Integer companyId);

}
