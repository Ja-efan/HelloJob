package com.ssafy.hellojob.global.auth.repository;

import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.global.auth.entity.Auth;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface AuthRepository extends JpaRepository<Auth, Integer> {

    Optional<Auth> findByUser(User user);
    boolean existsByUser(User user);
    void deleteByUser(User user);
}
