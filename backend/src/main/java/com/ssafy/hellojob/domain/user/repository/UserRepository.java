package com.ssafy.hellojob.domain.user.repository;

import com.ssafy.hellojob.domain.user.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Modifying;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.annotation.Transactional;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Integer> {

    @Modifying
    @Transactional
    @Query("UPDATE User u SET u.token = :tokenValue")
    void resetAllTokens(@Param("tokenValue") Integer tokenValue);

    Optional<User> findByEmail(String email);

}
