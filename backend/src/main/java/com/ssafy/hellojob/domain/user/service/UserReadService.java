package com.ssafy.hellojob.domain.user.service;

import com.ssafy.hellojob.domain.user.entity.User;
import com.ssafy.hellojob.domain.user.repository.UserRepository;
import com.ssafy.hellojob.global.exception.BaseException;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import static com.ssafy.hellojob.global.exception.ErrorCode.USER_NOT_FOUND;

@Service
@RequiredArgsConstructor
public class UserReadService {

    private final UserRepository userRepository;


    public User findUserByEmailOrElseThrow(String email) {
        return userRepository.findByEmail(email)
                .orElseThrow(() -> new BaseException(USER_NOT_FOUND));
    }

    public User findUserByIdOrElseThrow(Integer userId) {
        return userRepository.findById(userId)
                .orElseThrow(() -> new BaseException(USER_NOT_FOUND));
    }


}
