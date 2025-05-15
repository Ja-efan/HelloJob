
package com.ssafy.hellojob.domain.company.entity;

import jakarta.persistence.*;
import java.time.LocalDateTime;

import com.ssafy.hellojob.domain.company.entity.CompanySize;
import lombok.Getter;

@Entity
@Getter
@Table(name = "company")
public class Company {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "company_id", nullable = false)
    private Integer companyId;

    @Column(name = "company_name", nullable = false)
    private String companyName;

    @Column(name = "company_location")
    private String companyLocation;

    @Enumerated(EnumType.STRING)
    @Column(name = "company_size", nullable = false)
    private CompanySize companySize;

    @Column(name = "company_industry")
    private String companyIndustry;

    @Column(name = "company_visible", nullable = false)
    private boolean companyVisible;

    @Column(name = "updated_at", nullable = false)
    private LocalDateTime updatedAt;

    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
}
