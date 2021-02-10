/*
 * Copyright 2017-2020 CNRS-AIST JRL and CNRS-UM LIRMM
 */

#pragma once

#include <tvm/defs.h>

#include <tvm/graph/abstract/Node.h>
#include <tvm/task_dynamics/abstract/TaskDynamicsImpl.h>
#include <tvm/task_dynamics/enums.h>

#include <Eigen/Core>

// FIXME add mechanisms for when the function's output is resized
// FIXME Consider the possibility of having variables in task dynamics?

namespace tvm
{

namespace task_dynamics
{

namespace abstract
{
/** This is a base class to describe how a task is to be regulated, i.e. how
 * to compute e^(d)* for a task with constraint part f op rhs, where f is a
 * function, op is one operator among (==, <=, >=), rhs is a constant or a
 * vector and e = f-rhs. d is the order of the task dynamics.
 *
 * TaskDynamics is a lightweight descriptor, independent of a particular
 * task, that is meant for the end user.
 * Internally, it is turned into a TaskDynamicsImpl when linked to a given
 * function and rhs.
 */
class TVM_DLLAPI TaskDynamics
{
public:
  virtual ~TaskDynamics() = default;

  std::unique_ptr<TaskDynamicsImpl> impl(FunctionPtr f, constraint::Type t, const Eigen::VectorXd & rhs) const;

  Order order() const;

protected:
  virtual std::unique_ptr<TaskDynamicsImpl> impl_(FunctionPtr f,
                                                  constraint::Type t,
                                                  const Eigen::VectorXd & rhs) const = 0;
  virtual Order order_() const = 0;
};

} // namespace abstract

} // namespace task_dynamics

} // namespace tvm

// Workaround to avoid relying on non-standard code, here ##__VA_ARGS__.
// Works up to 5 arguments, adapted from https://stackoverflow.com/a/35214790
// Disclaimer: I don't understand everything here. To support more arguments, add some aXY to TVM_CALL and TVM_CALLN to
// TVM_CHOOSE until it works
#define TVM_COMMA_IF_PARENS(...) ,
#define TVM_LPAREN (
#define TVM_EXPAND(...) __VA_ARGS__
#define TVM_CHOOSE(...)                               \
    TVM_EXPAND(TVM_CALL TVM_LPAREN \
      __VA_ARGS__ TVM_COMMA_IF_PARENS \
      __VA_ARGS__ TVM_COMMA_IF_PARENS __VA_ARGS__ (), \
      TVM_CALLN, impossible, TVM_CALLN, TVM_CALLN, TVM_CALLN, TVM_CALLN, TVM_CALLN, TVM_CALLN, TVM_CALLN, TVM_CALLN, TVM_CALLN, TVM_CALL0, TVM_CALLN, ))
#define TVM_CALL(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, arg, ...) arg
#define TVM_CALL0(x)
#define TVM_CALLN(...) , __VA_ARGS__
#define TVM_VA_ARGS(...) TVM_CHOOSE(__VA_ARGS__)(__VA_ARGS__)

/** This macro can be used to define the derived factory required in
 * TaskDynamics implementation, \p Args are the arguments required by the derived
 * class, the macro arguments are members of the class passed to the derived
 * constructor */
#define TASK_DYNAMICS_DERIVED_FACTORY(...)                                                             \
  template<typename Derived, typename... Args>                                                         \
  std::unique_ptr<tvm::task_dynamics::abstract::TaskDynamicsImpl> impl_(                               \
      tvm::FunctionPtr f, tvm::constraint::Type t, const Eigen::VectorXd & rhs, Args &&... args) const \
  {                                                                                                    \
    return std::make_unique<Derived>(f, t, rhs, std::forward<Args>(args)... TVM_VA_ARGS(__VA_ARGS__)); \
  }

/** This macro can be used to define the derived factory required in composable
 * TaskDynamics implementation, \p Args are the arguments required by the derived
 * class, the macro variadic arguments are members of the class passed to the
 * derived constructor, the first argument is the template argument
 * representing the encapsulated TaskDynamic type */
#define COMPOSABLE_TASK_DYNAMICS_DERIVED_FACTORY(T, ...)                                                \
  template<typename Derived, typename... Args>                                                          \
  std::unique_ptr<tvm::task_dynamics::abstract::TaskDynamicsImpl> impl_(                                \
      tvm::FunctionPtr f, tvm::constraint::Type t, const Eigen::VectorXd & rhs, Args &&... args) const  \
  {                                                                                                     \
    return T::template impl_<Derived>(f, t, rhs, std::forward<Args>(args)... TVM_VA_ARGS(__VA_ARGS__)); \
  }
